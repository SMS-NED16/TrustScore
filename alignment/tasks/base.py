"""
Abstract base for alignment tasks: load samples, get human scores, and split data.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional


class AlignmentTask(ABC):
    """Task-specific dataset and human score interface for alignment."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Task identifier (e.g. 'summeval')."""
        pass

    @abstractmethod
    def load_samples(
        self,
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Load samples. Each dict must have at least:
        - prompt, response, sample_id or unique_dataset_id
        - any task-specific fields (e.g. expert_annotations, turker_annotations).
        """
        pass

    @abstractmethod
    def get_human_score(self, sample: Dict[str, Any]) -> float:
        """Return a single numeric target per sample. Raise if missing."""
        pass

    def get_splits(
        self,
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Deterministic train/val/test split. Default: 60% train, 20% val, 20% test.
        Override to split by article ID or unique_dataset_id to avoid leakage.
        """
        import random
        random.seed(random_seed)
        n = len(samples)
        indices = list(range(n))
        random.shuffle(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
        train = [samples[i] for i in train_idx]
        val = [samples[i] for i in val_idx]
        test = [samples[i] for i in test_idx]
        return train, val, test
