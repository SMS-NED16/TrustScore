"""
Load and sample datasets for specificity analysis
"""

import json
import os
from typing import List, Dict, Any, Optional
from scripts.load_summeval import load_summeval_with_sources


def load_and_sample_dataset(
    dataset_name: str = "summeval",
    max_samples: Optional[int] = None,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load and sample observations from SummEval or CNN/DailyMail dataset.
    
    Args:
        dataset_name: Name of dataset ("summeval" or "cnn_dailymail")
        max_samples: Maximum number of samples to load (None for all)
        random_seed: Random seed for reproducibility
        
    Returns:
        List of samples with prompt, response, and metadata
    """
    import random
    random.seed(random_seed)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    if dataset_name.lower() == "summeval":
        jsonl_path = os.path.join(
            project_root, 
            "datasets", 
            "raw", 
            "summeval", 
            "model_annotations.aligned.jsonl"
        )
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"SummEval file not found: {jsonl_path}")
        
        # Load SummEval data with sources
        summeval_data = load_summeval_with_sources(jsonl_path, max_samples=max_samples)
        
        # Convert to TrustScore format
        samples = []
        for sample in summeval_data:
            source_article = sample.get("source_article", "")
            summary = sample.get("summary", "")
            
            if source_article:
                prompt = f"Summarize the following article:\n\n{source_article}"
            else:
                prompt = "Generate a summary of the article."
            
            samples.append({
                "sample_id": sample.get("id", ""),
                "prompt": prompt,
                "response": summary,
                "model": sample.get("model_id", "unknown"),
                "source_article": source_article,
                "has_source": len(source_article) > 0,
                "metadata": {
                    "filepath": sample.get("filepath", ""),
                    "references": sample.get("references", []),
                }
            })
        
        # Randomly sample if needed
        if max_samples and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        return samples
    
    elif dataset_name.lower() == "cnn_dailymail":
        # TODO: Implement CNN/DailyMail loading if needed
        raise NotImplementedError("CNN/DailyMail loading not yet implemented")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def save_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Save samples to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    # Test loading
    samples = load_and_sample_dataset(max_samples=10)
    print(f"Loaded {len(samples)} samples")
    if samples:
        print(f"First sample: {samples[0]['sample_id']}")
