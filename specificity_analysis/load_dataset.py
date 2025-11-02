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
    
    # Find project root more reliably
    # Try current working directory first (useful for Colab)
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're already in project root
    possible_roots = [
        current_dir,  # Current working directory (Colab)
        os.path.dirname(script_dir),  # Parent of specificity_analysis
        os.path.dirname(os.path.dirname(script_dir)),  # Parent of parent
    ]
    
    # Find the one that has 'datasets' directory
    project_root = None
    for root in possible_roots:
        datasets_path = os.path.join(root, "datasets")
        if os.path.exists(datasets_path):
            project_root = root
            break
    
    if project_root is None:
        raise FileNotFoundError(
            f"Could not find project root. Tried: {possible_roots}\n"
            f"Make sure you're in the TrustScore directory and datasets/raw/summeval/ exists"
        )
    
    if dataset_name.lower() == "summeval":
        jsonl_path = os.path.join(
            project_root, 
            "datasets", 
            "raw", 
            "summeval", 
            "model_annotations.aligned.jsonl"
        )
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"SummEval file not found: {jsonl_path}\n"
                f"Project root: {project_root}\n"
                f"Make sure the dataset is downloaded to: {os.path.dirname(jsonl_path)}\n"
                f"Current working directory: {current_dir}"
            )
        
        # Load SummEval data with sources (load ALL to ensure diversity across articles)
        # Don't limit at this stage - we'll sample later to ensure diverse articles
        print(f"Loading all SummEval samples to ensure article diversity...")
        summeval_data = load_summeval_with_sources(jsonl_path, max_samples=None)  # Load all
        print(f"Loaded {len(summeval_data)} total samples from dataset")
        
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
                "sample_id": sample.get("id", ""),  # Original article ID
                "unique_dataset_id": sample.get("unique_dataset_id", ""),  # Unique identifier (article_id + model_id)
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
        
        # Randomly sample across different articles to ensure diversity
        if max_samples and len(samples) > max_samples:
            from collections import defaultdict
            
            # Group by article ID (sample_id) to understand distribution
            article_groups = defaultdict(list)
            for sample in samples:
                article_id = sample.get("sample_id", "")
                article_groups[article_id].append(sample)
            
            print(f"Found {len(article_groups)} unique articles in dataset")
            print(f"Attempting to sample {max_samples} samples across different articles...")
            
            # Strategy: Sample one summary per article to ensure diverse articles
            if len(article_groups) >= max_samples:
                # We have enough articles - sample one summary per article
                sampled_article_ids = random.sample(list(article_groups.keys()), max_samples)
                selected_samples = []
                for article_id in sampled_article_ids:
                    # Pick one random sample from this article's summaries
                    article_samples = article_groups[article_id]
                    selected_samples.append(random.choice(article_samples))
                samples = selected_samples
                print(f"✓ Sampled {len(samples)} samples from {len(sampled_article_ids)} different articles")
            else:
                # Not enough unique articles - sample what we can across articles, then fill remainder
                print(f"⚠ Only {len(article_groups)} unique articles available (requested {max_samples})")
                sampled_article_ids = list(article_groups.keys())  # Use all articles
                selected_samples = []
                
                # First, get one sample from each article
                for article_id in sampled_article_ids:
                    article_samples = article_groups[article_id]
                    selected_samples.append(random.choice(article_samples))
                
                # If we need more, randomly sample from remaining samples
                if len(selected_samples) < max_samples:
                    remaining_samples = [s for s in samples if s not in selected_samples]
                    additional_needed = max_samples - len(selected_samples)
                    if remaining_samples:
                        additional = random.sample(remaining_samples, min(additional_needed, len(remaining_samples)))
                        selected_samples.extend(additional)
                
                samples = selected_samples[:max_samples]
                print(f"✓ Sampled {len(samples)} samples ({len(sampled_article_ids)} unique articles)")
        
        # Verify diversity
        unique_article_ids = set(s.get("sample_id") for s in samples)
        unique_dataset_ids = set(s.get("unique_dataset_id") for s in samples)
        print(f"Final sample: {len(samples)} samples, {len(unique_article_ids)} unique articles, {len(unique_dataset_ids)} unique dataset IDs")
        
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
