"""
TopicalChat USR alignment task.

Data source: https://shikib.com/tc_usr_data.json
Schema (nested JSON array):
  [
    {
      "context": str,           -- dialogue history
      "fact":    str,           -- background knowledge
      "annotators": [str],      -- worker IDs
      "responses": [
        {
          "response": str,
          "model":    str,      -- e.g. "Original Ground Truth", "Argmax Decoding"
          "Overall":  [int, ...] -- one score per annotator (1-5 scale)
          ...                   -- other dimensions (Understandable, Natural, …) stored but unused
        }
      ]
    }
  ]

Flattening strategy:
  One sample per (conversation, response) pair.
  doc_id             = str(conversation_index)   -- used for split grouping
  unique_dataset_id  = "{conv_idx}-{model}"      -- row-unique, used as cache key
  sample_id          = same as unique_dataset_id

Human score: mean of the Overall annotation list for that response.

Splits: by doc_id so all model responses for the same conversation land in the same split.
"""

import json
import os
import random
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from alignment.tasks.base import AlignmentTask

SOURCE_URL = "https://shikib.com/tc_usr_data.json"


def _default_local_path() -> str:
    """Resolve datasets/raw/topicalchat_usr/tc_usr_data.json relative to project root."""
    here = os.path.dirname(os.path.abspath(__file__))
    # tasks/ -> alignment/ -> project root
    project_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(project_root, "datasets", "raw", "topicalchat_usr", "tc_usr_data.json")


def _ensure_downloaded(local_path: str) -> None:
    """Download the data file if it is not already present."""
    if os.path.exists(local_path):
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"[TopicalChatUSR] Downloading data from {SOURCE_URL} ...")
    urllib.request.urlretrieve(SOURCE_URL, local_path)
    print(f"[TopicalChatUSR] Saved to {local_path}")


def _mean_overall(response_obj: Dict[str, Any]) -> float:
    """Return mean of the Overall annotation list. Raises ValueError if no valid scores."""
    raw = response_obj.get("Overall", [])
    scores = []
    for v in raw:
        if v is None:
            continue
        try:
            f = float(v)
            if f == f:  # exclude NaN
                scores.append(f)
        except (TypeError, ValueError):
            pass
    if not scores:
        raise ValueError("No valid Overall scores found in response object")
    return sum(scores) / len(scores)


class TopicalChatUSRTask(AlignmentTask):
    """
    TopicalChat USR task: dialogue response quality evaluated by crowd workers.
    Human score = mean of Overall ratings (1–5 scale) across annotators.
    """

    def __init__(self, local_path: Optional[str] = None):
        self._local_path = local_path or _default_local_path()

    @property
    def name(self) -> str:
        return "topicalchat_usr"

    def load_samples(
        self,
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        _ensure_downloaded(self._local_path)

        with open(self._local_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples: List[Dict[str, Any]] = []
        for conv_idx, conv in enumerate(raw):
            context = (conv.get("context") or "").strip()
            fact = (conv.get("fact") or "").strip()
            responses = conv.get("responses") or []
            annotators = conv.get("annotators") or []

            if not context:
                continue

            # Build prompt once per conversation
            if fact:
                prompt = (
                    f"Given the following background information:\n\n{fact}"
                    f"\n\nContinue this conversation:\n\n{context}"
                )
            else:
                prompt = f"Continue this conversation:\n\n{context}"

            for resp_obj in responses:
                model = (resp_obj.get("model") or "unknown").strip()
                response_text = (resp_obj.get("response") or "").strip()

                if not response_text:
                    continue

                # Row-unique identifier used as cache key and in correlations.
                # Sanitise model name: spaces → underscores for readability, but
                # cache_path() in scoring.py will sanitise further if needed.
                safe_model = model.replace(" ", "_")
                unique_dataset_id = f"{conv_idx}-{safe_model}"

                sample = {
                    # Identifiers
                    "doc_id": str(conv_idx),           # conversation-level grouping for splits
                    "unique_dataset_id": unique_dataset_id,  # row-unique, used as cache key
                    "sample_id": unique_dataset_id,
                    # Content
                    "prompt": prompt,
                    "response": response_text,
                    "model": model,
                    # Raw annotations (stored for auditability; score computed lazily)
                    "overall_scores": resp_obj.get("Overall", []),
                    "annotators": annotators,
                    # Extra dimensions (not used in scoring but preserved for analysis)
                    "Understandable": resp_obj.get("Understandable", []),
                    "Natural": resp_obj.get("Natural", []),
                    "Maintains_Context": resp_obj.get("Maintains Context", []),
                    "Engaging": resp_obj.get("Engaging", []),
                    "Uses_Knowledge": resp_obj.get("Uses Knowledge", []),
                }

                # Skip rows where we cannot compute a valid human score
                try:
                    _mean_overall(resp_obj)
                except ValueError:
                    continue

                samples.append(sample)

        if max_samples is not None and len(samples) > max_samples:
            rng = random.Random(random_seed)
            samples = rng.sample(samples, max_samples)

        return samples

    def get_human_score(self, sample: Dict[str, Any]) -> float:
        """Mean of Overall ratings stored in the sample."""
        scores = []
        for v in sample.get("overall_scores", []):
            if v is None:
                continue
            try:
                f = float(v)
                if f == f:
                    scores.append(f)
            except (TypeError, ValueError):
                pass
        if not scores:
            raise ValueError(
                f"No valid Overall scores for sample {sample.get('unique_dataset_id')}"
            )
        return sum(scores) / len(scores)

    def get_splits(
        self,
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split by doc_id (conversation index) so all model responses for the
        same conversation never cross splits."""
        rng = random.Random(random_seed)

        # Collect unique doc_ids in stable insertion order
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

        def _split_key(s: Dict[str, Any]) -> str:
            return s.get("doc_id") or s.get("unique_dataset_id", "")

        train = [s for s in samples if _split_key(s) in train_ids]
        val   = [s for s in samples if _split_key(s) in val_ids]
        test  = [s for s in samples if _split_key(s) in test_ids]
        return train, val, test
