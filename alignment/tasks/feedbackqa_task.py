"""
FeedbackQA alignment task.

Data source: https://huggingface.co/datasets/McGill-NLP/feedbackQA
Loaded directly as JSON files (not via the datasets library).

Schema (one record per element in the JSON array):
  {
    "question":  str,
    "passage": {
      "passage_id": int,
      "source":     str,
      "url":        str,
      "reference": {
        "page_title":           str,
        "section_headers":      [str],
        "section_content":      str,   -- the answer text used as response
        "section_content_html": str,
        "selection_span":       any
      }
    },
    "feedback": [str, ...],   -- free-text annotator feedback (stored, not scored)
    "rating":   [str, ...],   -- e.g. ["Excellent", "Could be Improved"]
    "domain":   str
  }

Rating scale (RATING_MAP):
  Excellent        -> 4
  Good             -> 3
  Could be Improved -> 2
  Bad              -> 1

Human score: mean of numeric ratings across all annotators.

Splits: by passage_id so the same passage never appears in both train and test.
"""

import json
import os
import random
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from alignment.tasks.base import AlignmentTask

# ---------------------------------------------------------------------------
# Rating conversion
# ---------------------------------------------------------------------------

RATING_MAP: Dict[str, int] = {
    "Excellent": 4,
    "Good": 3,
    "Could be Improved": 2,
    "Bad": 1,
}

HF_BASE_URL = (
    "https://huggingface.co/datasets/McGill-NLP/feedbackQA/resolve/main/data"
)
SPLIT_FILES = {
    "train": "feedback_train.json",
    "validation": "feedback_valid.json",
    "test": "feedback_test.json",
}


def _default_cache_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(project_root, "datasets", "raw", "feedbackqa")


def _ensure_split_downloaded(split: str, cache_dir: str) -> str:
    """Download a single split JSON if not already cached. Returns local path."""
    filename = SPLIT_FILES[split]
    local_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_path):
        os.makedirs(cache_dir, exist_ok=True)
        url = f"{HF_BASE_URL}/{filename}"
        print(f"[FeedbackQA] Downloading {split} split from {url} ...")
        urllib.request.urlretrieve(url, local_path)
        print(f"[FeedbackQA] Saved to {local_path}")
    return local_path


def _load_split(split: str, cache_dir: str) -> List[Dict[str, Any]]:
    path = _ensure_split_downloaded(split, cache_dir)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ratings_to_scalar(ratings: List[str]) -> Optional[float]:
    scores = [RATING_MAP[r] for r in ratings if r in RATING_MAP]
    return sum(scores) / len(scores) if scores else None


def _get_section_content(record: Dict[str, Any]) -> str:
    """Extract answer text: passage.reference.section_content with fallbacks."""
    passage = record.get("passage") or {}
    # Primary path: passage -> reference -> section_content
    ref = passage.get("reference") or {}
    content = ref.get("section_content", "")
    if content:
        return content
    # Fallback: passage -> section_content (some dataset versions flatten this)
    return passage.get("section_content", "")


class FeedbackQATask(AlignmentTask):
    """
    FeedbackQA task: question-answering quality rated by human annotators
    on a 4-point scale (Bad / Could be Improved / Good / Excellent).
    Human score = mean numeric rating across annotators (1–4 scale).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir or _default_cache_dir()

    @property
    def name(self) -> str:
        return "feedbackqa"

    def load_samples(
        self,
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        # Load and combine all three splits so the alignment framework
        # controls train/val/test proportions via get_splits().
        raw: List[Dict[str, Any]] = []
        for split in ("train", "validation", "test"):
            raw.extend(_load_split(split, self._cache_dir))

        samples: List[Dict[str, Any]] = []
        for row_idx, record in enumerate(raw):
            question = (record.get("question") or "").strip()
            response = _get_section_content(record).strip()

            if not question or not response:
                continue

            human_score = _ratings_to_scalar(record.get("rating") or [])
            if human_score is None:
                continue

            passage = record.get("passage") or {}
            passage_id = passage.get("passage_id")
            doc_id = str(passage_id) if passage_id is not None else str(row_idx)

            # Row-unique identifier: passage_id + row index guards against
            # multiple questions referencing the same passage.
            unique_dataset_id = f"{doc_id}-{row_idx}"

            samples.append({
                # Identifiers
                "doc_id": doc_id,                      # passage-level grouping for splits
                "unique_dataset_id": unique_dataset_id, # row-unique cache key
                "sample_id": unique_dataset_id,
                # Content
                "prompt": question,
                "response": response,
                # Human annotation
                "human_score": human_score,
                "ratings": record.get("rating", []),
                "feedback": record.get("feedback", []),
                # Metadata
                "domain": record.get("domain", "unknown"),
                "url": passage.get("url", ""),
            })

        if max_samples is not None and len(samples) > max_samples:
            rng = random.Random(random_seed)
            samples = rng.sample(samples, max_samples)

        return samples

    def get_human_score(self, sample: Dict[str, Any]) -> float:
        score = sample.get("human_score")
        if score is None:
            raise ValueError(
                f"No human_score for sample {sample.get('unique_dataset_id')}"
            )
        return float(score)

    def get_splits(
        self,
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split by doc_id (passage_id) so the same passage never crosses splits."""
        rng = random.Random(random_seed)

        doc_ids = list(dict.fromkeys(
            s.get("doc_id") or s.get("unique_dataset_id", str(i))
            for i, s in enumerate(samples)
        ))
        rng.shuffle(doc_ids)

        n = len(doc_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train_ids = set(doc_ids[:n_train])
        val_ids = set(doc_ids[n_train: n_train + n_val])
        test_ids = set(doc_ids[n_train + n_val:])

        def _key(s: Dict[str, Any]) -> str:
            return s.get("doc_id") or s.get("unique_dataset_id", "")

        train = [s for s in samples if _key(s) in train_ids]
        val   = [s for s in samples if _key(s) in val_ids]
        test  = [s for s in samples if _key(s) in test_ids]
        return train, val, test
