"""
SummEval alignment task: load samples and derive human quality score from
expert_annotations and turker_annotations (coherence, consistency, fluency, relevance).
"""

import os
from typing import List, Dict, Any, Optional, Tuple

from alignment.tasks.base import AlignmentTask

# SummEval dimensions (from paper). Annotation format: list of dicts per annotator,
# each dict with keys like "coherence", "consistency", "fluency", "relevance" (scores typically 1-5).
SUMMEVAL_DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]


def _parse_summeval_annotations(sample: Dict[str, Any]) -> float:
    """
    Aggregate expert + turker annotations into a single human quality score.
    Each annotator dict may have keys: coherence, consistency, fluency, relevance (or similar).
    Returns mean over all dimensions and all annotators. Missing data raises ValueError.
    """
    expert = sample.get("expert_annotations") or []
    turker = sample.get("turker_annotations") or []
    all_annotations = list(expert) + list(turker)
    if not all_annotations:
        raise ValueError("Sample has no expert_annotations or turker_annotations")

    scores = []
    for ann in all_annotations:
        if not isinstance(ann, dict):
            continue
        for dim in SUMMEVAL_DIMENSIONS:
            # Allow slight key variants (e.g. first letter capitalized)
            v = ann.get(dim) or ann.get(dim.capitalize())
            if v is not None:
                try:
                    scores.append(float(v))
                except (TypeError, ValueError):
                    pass
    if not scores:
        raise ValueError(
            f"No numeric scores found in annotations; expected keys like {SUMMEVAL_DIMENSIONS}"
        )
    return sum(scores) / len(scores)


class SummEvalTask(AlignmentTask):
    """SummEval task: summarization quality (coherence, consistency, fluency, relevance)."""

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        project_root: Optional[str] = None,
    ):
        if jsonl_path:
            self._jsonl_path = jsonl_path
        else:
            root = project_root or _find_project_root()
            self._jsonl_path = os.path.join(
                root, "datasets", "raw", "summeval", "model_annotations.aligned.jsonl"
            )

    @property
    def name(self) -> str:
        return "summeval"

    def load_samples(
        self,
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        from scripts.load_summeval import load_summeval_with_sources

        if not os.path.exists(self._jsonl_path):
            raise FileNotFoundError(
                f"SummEval file not found: {self._jsonl_path}. "
                "Download model_annotations.aligned.jsonl to that path."
            )
        raw = load_summeval_with_sources(self._jsonl_path, max_samples=max_samples)
        samples = []
        for s in raw:
            source = s.get("source_article", "")
            summary = s.get("summary", "")
            prompt = (
                f"Summarize the following article:\n\n{source}"
                if source
                else "Generate a summary of the article."
            )
            sample = {
                "unique_dataset_id": s.get("unique_dataset_id", f"{s.get('id', '')}-{s.get('model_id', '')}"),
                "sample_id": s.get("id", ""),
                "prompt": prompt,
                "response": summary,
                "model": s.get("model_id", "unknown"),
                "expert_annotations": s.get("expert_annotations", []),
                "turker_annotations": s.get("turker_annotations", []),
                "source_article": source,
            }
            # Skip samples with no human annotations so get_human_score() never raises later
            try:
                _parse_summeval_annotations(sample)
            except ValueError:
                continue
            samples.append(sample)
        return samples

    def get_human_score(self, sample: Dict[str, Any]) -> float:
        return _parse_summeval_annotations(sample)

    def get_splits(
        self,
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split by unique_dataset_id so same article+model never crosses splits."""
        import random
        random.seed(random_seed)
        ids = [s.get("unique_dataset_id") or s.get("sample_id", str(i)) for i, s in enumerate(samples)]
        unique_ids = list(dict.fromkeys(ids))
        random.shuffle(unique_ids)
        n = len(unique_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        train_ids = set(unique_ids[:n_train])
        val_ids = set(unique_ids[n_train : n_train + n_val])
        test_ids = set(unique_ids[n_train + n_val :])
        train = [s for s in samples if (s.get("unique_dataset_id") or s.get("sample_id", "")) in train_ids]
        val = [s for s in samples if (s.get("unique_dataset_id") or s.get("sample_id", "")) in val_ids]
        test = [s for s in samples if (s.get("unique_dataset_id") or s.get("sample_id", "")) in test_ids]
        return train, val, test


def _find_project_root() -> str:
    current = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for root in [current, os.path.dirname(script_dir), os.path.dirname(os.path.dirname(script_dir))]:
        if os.path.exists(os.path.join(root, "datasets")):
            return root
    return current
