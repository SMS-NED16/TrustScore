"""
SimpEval alignment task: load samples from tasksource/simpeval on HuggingFace and
derive human quality score as mean of adequacy, fluency, and simplicity annotations.

Only machine-generated system outputs are included (Human 1 Writing and Human 2 Writing
rows are excluded). The unique_dataset_id is the document id so that all system outputs
for a given document land in the same split.
"""

import random
from typing import List, Dict, Any, Optional, Tuple

from alignment.tasks.base import AlignmentTask

# System values to exclude (human-written references, not model outputs)
HUMAN_SYSTEMS = {"Human 1 Writing", "Human 2 Writing"}

# Score columns to average into a single human quality score (0-100 scale)
SIMPEVAL_SCORE_COLUMNS = ["Answer.adequacy", "Answer.fluency", "Answer.simplicity"]


def _mean_score(sample: Dict[str, Any]) -> float:
    """Average the three annotation dimensions into a single quality score."""
    scores = []
    for col in SIMPEVAL_SCORE_COLUMNS:
        v = sample.get(col)
        if v is not None:
            try:
                f = float(v)
                if f == f:  # exclude NaN
                    scores.append(f)
            except (TypeError, ValueError):
                pass
    if not scores:
        raise ValueError(
            f"No valid scores found; expected keys {SIMPEVAL_SCORE_COLUMNS}"
        )
    return sum(scores) / len(scores)


class SimpEvalTask(AlignmentTask):
    """SimpEval task: text simplification quality (adequacy, fluency, simplicity)."""

    def __init__(self, hf_dataset_name: str = "tasksource/simpeval"):
        self._hf_dataset_name = hf_dataset_name

    @property
    def name(self) -> str:
        return "simpeval"

    def load_samples(
        self,
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(self._hf_dataset_name, split="train")

        samples = []
        for row in ds:
            # Filter out human-written references
            system = row.get("Input.system", "")
            if system in HUMAN_SYSTEMS:
                continue

            original = row.get("Input.original", "") or ""
            simplified = row.get("Input.simplified", "") or ""

            if not original or not simplified:
                continue

            doc_id = row.get("id", "")
            sample_id = f"{doc_id}-{system}"

            sample = {
                "unique_dataset_id": str(doc_id),
                "sample_id": sample_id,
                "prompt": f"Simplify the following text:\n\n{original}",
                "response": simplified,
                "model": system,
                "original_text": original,
                "Answer.adequacy": row.get("Answer.adequacy"),
                "Answer.fluency": row.get("Answer.fluency"),
                "Answer.simplicity": row.get("Answer.simplicity"),
            }

            # Skip rows where we can't compute a valid score
            try:
                _mean_score(sample)
            except ValueError:
                continue

            samples.append(sample)

        if max_samples is not None and len(samples) > max_samples:
            rng = random.Random(random_seed)
            samples = rng.sample(samples, max_samples)

        return samples

    def get_human_score(self, sample: Dict[str, Any]) -> float:
        return _mean_score(sample)

    def get_splits(
        self,
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split by unique_dataset_id (document id) so all system outputs for a
        given document never cross splits."""
        rng = random.Random(random_seed)
        ids = [s.get("unique_dataset_id") or s.get("sample_id", str(i)) for i, s in enumerate(samples)]
        unique_ids = list(dict.fromkeys(ids))
        rng.shuffle(unique_ids)
        n = len(unique_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        train_ids = set(unique_ids[:n_train])
        val_ids = set(unique_ids[n_train: n_train + n_val])
        test_ids = set(unique_ids[n_train + n_val:])
        train = [s for s in samples if (s.get("unique_dataset_id") or s.get("sample_id", "")) in train_ids]
        val = [s for s in samples if (s.get("unique_dataset_id") or s.get("sample_id", "")) in val_ids]
        test = [s for s in samples if (s.get("unique_dataset_id") or s.get("sample_id", "")) in test_ids]
        return train, val, test
