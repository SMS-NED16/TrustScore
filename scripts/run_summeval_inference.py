"""
Run TrustScore inference on SummEval dataset
"""

import json
import os
import time
from datetime import datetime
from tqdm import tqdm
from pipeline.orchestrator import TrustScorePipeline
from config.settings import TrustScoreConfig, JudgeConfig, LLMConfig, LLMProvider, SpanTaggerConfig, AggregationWeights


def run_summeval_inference(
    input_file: str,
    output_file: str,
    max_samples: int = None,
    batch_size: int = 10,
    use_vllm: bool = True,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
):
    """
    Run TrustScore inference on preprocessed SummEval data.
    
    Args:
        input_file: Path to preprocessed JSONL file
        output_file: Path to save results
        max_samples: Maximum number of samples to process
        batch_size: Batch size for processing
        use_vllm: Whether to use vLLM provider
        model: Model name to use (default: LLaMA)
    """
    print("=" * 70)
    print("SummEval TrustScore Inference")
    print("=" * 70)
    
    # Load preprocessed data
    print(f"\nLoading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} samples")
    
    # Initialize pipeline
    print("\nInitializing TrustScore pipeline...")
    
    if use_vllm:
        # Configure vLLM
        llm_config = SpanTaggerConfig(
            provider=LLMProvider.VLLM,
            model=model,
            temperature=0.1,
            max_tokens=2000,
            batch_size=batch_size,
        )
        print(f"✅ Using vLLM provider with model: {model}")
    else:
        # Configure OpenAI (fallback)
        llm_config = SpanTaggerConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            temperature=0.1,
            max_tokens=2000,
        )
        print("✅ Using OpenAI provider")
    
    # Create judge configs
    judge_configs = {
        "trustworthiness_judge": JudgeConfig(
            name="trustworthiness_judge",
            enabled=True,
            provider=llm_config.provider,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        ),
        "bias_judge": JudgeConfig(
            name="bias_judge",
            enabled=True,
            provider=llm_config.provider,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        ),
        "explainability_judge": JudgeConfig(
            name="explainability_judge",
            enabled=True,
            provider=llm_config.provider,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        ),
    }
    
    # Create config with span tagger config
    config = TrustScoreConfig(
        span_tagger=llm_config,
        judges=judge_configs,
        aggregation_weights=AggregationWeights(
            trustworthiness=0.6,
            explainability=0.3,
            bias=0.1
        )
    )
    
    # Initialize pipeline
    pipeline = TrustScorePipeline(config=config, use_mock=False)
    
    # Process samples
    print(f"\nProcessing {len(data)} samples in batches of {batch_size}...")
    print(f"Using provider: {llm_config.provider.value}")
    
    results = []
    start_time = time.time()
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i+batch_size]
        
        batch_inputs = []
        for sample in batch:
            batch_inputs.append({
                'prompt': sample['prompt'],
                'response': sample['response'],
                'model': sample['model'],
                'generated_on': datetime.fromisoformat(sample['generated_on']),
            })
        
        try:
            batch_results = pipeline.process_batch(batch_inputs)
            
            # Combine results with original data
            for j, result in enumerate(batch_results):
                if result:
                    output_record = {
                        "sample_id": batch[j].get("sample_id", f"sample_{i+j}"),
                        "timestamp": datetime.now().isoformat(),
                        "trustscore_output": result.format_for_output(config.output),
                        "original_annotations": {
                            "expert": batch[j].get("expert_annotations", []),
                            "turker": batch[j].get("turker_annotations", []),
                            "references": batch[j].get("references", []),
                        },
                    }
                    results.append(output_record)
                else:
                    print(f"⚠️  Failed to process sample {i+j}")
                    
        except Exception as e:
            print(f"❌ Error processing batch {i//batch_size}: {e}")
            # Add empty results for failed batch
            for j in range(len(batch)):
                results.append({
                    "sample_id": batch[j].get("sample_id", f"sample_{i+j}"),
                    "error": str(e),
                })
    
    elapsed_time = time.time() - start_time
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Inference Complete")
    print("=" * 70)
    print(f"Total samples processed: {len(results)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    if len(results) > 0:
        print(f"Average time per sample: {elapsed_time/len(results):.2f} seconds")
        print(f"Throughput: {len(results)/elapsed_time:.2f} samples/second")
    else:
        print("⚠️  No samples were successfully processed")
    
    # Calculate statistics
    successful_results = [r for r in results if 'error' not in r and 'trustscore_output' in r]
    
    if successful_results:
        print(f"\nSuccessful analyses: {len(successful_results)}")
        
        # Extract trust scores
        trust_scores = []
        for r in successful_results:
            try:
                ts = r['trustscore_output']['summary']['trust_score']
                trust_scores.append(ts)
            except:
                pass
        
        if trust_scores:
            print(f"\nTrust Score Statistics:")
            print(f"  Mean: {sum(trust_scores)/len(trust_scores):.3f}")
            print(f"  Min: {min(trust_scores):.3f}")
            print(f"  Max: {max(trust_scores):.3f}")
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_root, "datasets", "processed", "summeval_trustscore_format.jsonl")
    output_file = os.path.join(project_root, "results", "summeval_trustscore_100samples.jsonl")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("   Please run preprocess_summeval.py first")
        exit(1)
    
    run_summeval_inference(
        input_file=input_file,
        output_file=output_file,
        max_samples=100,
        batch_size=10,
        use_vllm=True,
        model="meta-llama/Llama-3.1-8B-Instruct",
    )

