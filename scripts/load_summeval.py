"""
Load SummEval dataset with CNN/DailyMail source articles
"""

import json
import os
from typing import Dict, Any, List
from datasets import load_dataset


def load_summeval_with_sources(jsonl_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """
    Load SummEval dataset with source articles from CNN/DailyMail.
    
    Args:
        jsonl_path: Path to SummEval JSONL file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of samples with source articles attached
    """
    print("Loading CNN/DailyMail dataset from HuggingFace...")
    
    # Load the CNN/DailyMail dataset for source articles
    try:
        cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="test")
        print(f"[Info] Loaded {len(cnn_dm)} CNN/DailyMail articles")
    except Exception as e:
        print(f"[Warning] Could not load CNN/DailyMail dataset: {e}")
        print("   Will try to match source articles by filepath...")
        cnn_dm = None
    
    # Create a mapping from filepath to article
    article_map = {}
    if cnn_dm:
        for article in cnn_dm:
            filepath = article.get("id", "")
            article_map[filepath] = article
    
    print(f"Loaded {len(article_map)} articles into mapping")
    
    # Load SummEval annotations
    print(f"\nLoading SummEval annotations from {jsonl_path}...")
    summeval_data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                data = json.loads(line.strip())
                
                # Extract fields
                sample_id = data.get("id", f"sample_{i}")
                summary = data.get("decoded", "")
                model_id = data.get("model_id", "unknown")
                filepath = data.get("filepath", "")
                
                # Get expert annotations
                expert_annotations = data.get("expert_annotations", [])
                
                # Get turker annotations
                turker_annotations = data.get("turker_annotations", [])
                
                # Get references
                references = data.get("references", [])
                
                # Try to find source article
                source_article = None
                source_text = ""
                
                if article_map:
                    # Extract ID from filepath by removing directory and .story extension
                    filepath_base = filepath.split("/")[-1]  # Get filename
                    file_id = filepath_base.replace(".story", "")  # Remove .story extension
                    
                    # Try different path variations
                    possible_paths = [
                        file_id,  # Just the ID (most likely to match)
                        filepath_base,  # Filename with .story
                        filepath,  # Full path
                        filepath.replace("stories/", ""),
                        filepath.replace("cnndm/dailymail/stories/", ""),
                    ]
                    
                    for path in possible_paths:
                        if path in article_map:
                            source_article = article_map[path]
                            source_text = source_article.get("article", "")
                            break
                
                sample = {
                    "id": sample_id,
                    "summary": summary,
                    "model_id": model_id,
                    "filepath": filepath,
                    "source_article": source_text,
                    "references": references,
                    "expert_annotations": expert_annotations,
                    "turker_annotations": turker_annotations,
                    "has_source": source_text != ""
                }
                
                summeval_data.append(sample)
                
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipped malformed line {i}: {e}")
                continue
    
    print(f"[Info] Loaded {len(summeval_data)} SummEval samples")
    print(f"   Samples with source articles: {sum([1 for s in summeval_data if s['has_source']])}")
    
    return summeval_data


def print_sample_statistics(data: List[Dict[str, Any]]):
    """Print statistics about the loaded dataset"""
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(data)}")
    print(f"Samples with source articles: {sum([1 for s in data if s['has_source']])}")
    
    # Count by model
    model_counts = {}
    for sample in data:
        model_id = sample.get("model_id", "unknown")
        model_counts[model_id] = model_counts.get(model_id, 0) + 1
    
    print(f"\nSamples by model:")
    for model_id, count in sorted(model_counts.items()):
        print(f"  {model_id}: {count}")
    
    # Check for source articles
    if data and data[0].get("has_source"):
        print("\n[Info] Source articles available for TrustScore analysis")
    else:
        print("\n[Warning] No source articles found - will use summaries only")


if __name__ == "__main__":
    # Test loading with 10 samples
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    jsonl_path = os.path.join(project_root, "datasets", "raw", "summeval", "model_annotations.aligned.jsonl")
    
    if not os.path.exists(jsonl_path):
        print(f"[Error] File not found: {jsonl_path}")
        exit(1)
    
    data = load_summeval_with_sources(jsonl_path, max_samples=10)
    print_sample_statistics(data)
    
    # Print first sample
    if data:
        print("\n=== First Sample ===")
        sample = data[0]
        print(f"ID: {sample['id']}")
        print(f"Model: {sample['model_id']}")
        print(f"Has source: {sample['has_source']}")
        print(f"Summary length: {len(sample['summary'])} chars")
        print(f"Source length: {len(sample['source_article'])} chars")

