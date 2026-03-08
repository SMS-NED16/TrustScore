"""
Base ingestion adapter for the TrustScore Recommender.

All adapters share a common JSON loading utility and must produce
List[RankRecord] from a file path.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from recommender.schemas import RankRecord

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """Abstract base for leaderboard ingestion adapters."""

    @abstractmethod
    def load(self, path: Path) -> List[RankRecord]:
        """Load rank records from the given file path."""
        ...

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        """Read and parse a JSON file, returning an empty dict on failure."""
        if not path.exists():
            logger.warning("Rank file does not exist: %s", path)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read rank file %s: %s", path, exc)
            return {}

    @staticmethod
    def _entries_to_records(
        entries: List[Dict[str, Any]],
        source_name: str,
        benchmark_name: str,
        source_url: str = "",
        snapshot_date: str = "",
    ) -> List[RankRecord]:
        """
        Convert a list of raw JSON entries to RankRecord objects.

        Each entry is expected to have at least 'model' and 'rank' keys.
        """
        records: List[RankRecord] = []
        for entry in entries:
            model_raw = entry.get("model", "")
            rank = entry.get("rank")
            if not model_raw or rank is None:
                logger.warning("Skipping malformed entry in %s: %s", benchmark_name, entry)
                continue
            records.append(RankRecord(
                source_name=source_name,
                benchmark_name=benchmark_name,
                model_name_raw=model_raw,
                rank=int(rank),
                score=entry.get("score"),
                source_url=source_url,
                snapshot_date=snapshot_date,
                metadata={k: v for k, v in entry.get("metadata", {}).items()},
            ))
        return records
