"""
Preprocess SummEval dataset to TrustScore format
"""

import json
import os
from datetime import datetime
from scripts.load_summeval import load_summeval_with_sources


def preprocess_to_trustscore_format(max_samples: int = 100):
    """
    Preprocess SummEval data to TrustScore format.
    
    Args:
        max_samples: Number of samples to process
    """
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    jsonl_path = os.path.join(project_root, "datasets", "raw", "summeval", "model_annotations.aligned.jsonl")
    output_path = os.path.join(project_root, "datasets", "processed", "summeval_trustscore_format.jsonl")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 70)
    print("SummEval to TrustScore Preprocessing")
    print("=" * 70)
    
    # Load SummEval data
    summeval_data = load_summeval_with_sources(jsonl_path, max_samples=max_samples)
    
    # Convert to TrustScore format
    trustscore_data = []
    
    for i, sample in enumerate(summeval_data):
        # Extract fields
        source_article = sample.get("source_article", "")
        summary = sample.get("summary", "")
        model_id = sample.get("model_id", "unknown")
        sample_id = sample.get("id", f"sample_{i}")
        
        # Create prompt
        if source_article:
            prompt = f"Summarize the following article:\n\n{source_article}"
        else:
            # Fallback if no source article
            prompt = "Generate a summary of the article."
        
        # Create TrustScore format record
        trustscore_record = {
            "prompt": prompt,
            "response": summary,
            "model": model_id,
            "generated_on": datetime.now().isoformat(),
            "sample_id": sample_id,
            "source_article": source_article,
            "has_source": len(source_article) > 0,
            "references": sample.get("references", []),
            "expert_annotations": sample.get("expert_annotations", []),
            "turker_annotations": sample.get("turker_annotations", []),
        }
        
        trustscore_data.append(trustscore_record)
    
    # Save to JSONL
    print(f"\nSaving preprocessed data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in trustscore_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(trustscore_data)} samples to {output_path}")
    
    # Print statistics
    print("\n=== Preprocessing Statistics ===")
    print(f"Total samples: {len(trustscore_data)}")
    print(f"Samples with source articles: {sum([1 for r in trustscore_data if r['has_source']])}")
    print(f"Samples without source articles: {sum([1 for r in trustscore_data if not r['has_source']])}")
    
    # Show sample
    if trustscore_data:
        print("\n=== Sample Record ===")
        sample = trustscore_data[0]
        print(f"Sample ID: {sample['sample_id']}")
        print(f"Model: {sample['model']}")
        print(f"Has source: {sample['has_source']}")
        print(f"Prompt length: {len(sample['prompt'])} chars")
        print(f"Response length: {len(sample['response'])} chars")
        print(f"\nResponse preview:")
        print(sample['response'][:200] + "...")
    
    return output_path


if __name__ == "__main__":
    # Preprocess 100 samples
    output_path = preprocess_to_trustscore_format(max_samples=100)
    print(f"\n✅ Preprocessing complete! Output: {output_path}")

