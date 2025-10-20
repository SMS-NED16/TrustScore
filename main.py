"""
TrustScore Pipeline - Main Entry Point

This module provides the main entry point for the TrustScore pipeline
and demonstrates the complete workflow.
"""

from datetime import datetime
from pipeline.orchestrator import TrustScorePipeline, analyze_llm_response
from config.settings import load_config
from examples.usage import run_all_examples


def main() -> None:
    """Main entry point for TrustScore pipeline."""
    print("TrustScore Pipeline - Main Entry Point")
    print("=" * 50)
    
    # Example 1: Quick analysis
    print("\n1. Quick Analysis Example:")
    try:
        result: AggregatedOutput = analyze_llm_response(
            prompt="When was Georgia Tech founded?",
            response="Georgia Institute of Technology was founded in 1885. It's located in Atlanta, Georgia.",
            model="GPT-4o",
            use_mock=True
        )
        
        print(f"   Trust Score: {result.summary.trust_score}")
        print(f"   Trustworthiness: {result.summary.agg_score_T}")
        print(f"   Explainability: {result.summary.agg_score_E}")
        print(f"   Bias: {result.summary.agg_score_B}")
        print(f"   Errors found: {len(result.errors)}")
        
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Example 2: Full pipeline with custom config
    print("\n2. Full Pipeline Example:")
    try:
        config: TrustScoreConfig = load_config()
        config.aggregation_weights.trustworthiness = 0.7
        config.aggregation_weights.explainability = 0.2
        config.aggregation_weights.bias = 0.1
        
        pipeline: TrustScorePipeline = TrustScorePipeline(config=config, use_mock=True)
        
        result: AggregatedOutput = pipeline.process(
            prompt="Explain machine learning",
            response="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            model="GPT-4o"
        )
        
        print(f"   Trust Score: {result.summary.trust_score}")
        print(f"   Confidence Interval: [{result.summary.trust_score_ci.lower}, {result.summary.trust_score_ci.upper}]")
        
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Example 3: Batch processing
    print("\n3. Batch Processing Example:")
    try:
        pipeline: TrustScorePipeline = TrustScorePipeline(use_mock=True)
        
        batch_inputs: List[Dict[str, Any]] = [
            {
                "prompt": "What is Python?",
                "response": "Python is a high-level programming language known for its simplicity and readability.",
                "model": "GPT-4o"
            },
            {
                "prompt": "How does photosynthesis work?",
                "response": "Photosynthesis is the process by which plants convert sunlight into energy.",
                "model": "GPT-4o"
            }
        ]
        
        results: List[AggregatedOutput] = pipeline.process_batch(batch_inputs)
        
        for i, result in enumerate(results):
            if result:
                print(f"   Batch item {i+1}: Trust Score = {result.summary.trust_score}")
            else:
                print(f"   Batch item {i+1}: Processing failed")
                
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Main examples completed!")
    
    # Optionally run all detailed examples
    print("\nWould you like to run detailed examples? (y/n): ", end="")
    try:
        response: str = input().lower().strip()
        if response in ['y', 'yes']:
            print("\nRunning detailed examples...")
            run_all_examples()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
