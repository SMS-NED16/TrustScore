"""
Ingestion adapter for ProLLM LLM-as-a-Judge leaderboard snapshots.

Gracefully handles empty data (the current state of the ProLLM leaderboard).
"""

import logging
from pathlib import Path
from typing import List

from recommender.ingestion.base import BaseAdapter
from recommender.schemas import RankRecord

logger = logging.getLogger(__name__)


class ProllmAdapter(BaseAdapter):
    """Load rank records from the ProLLM JSON snapshot (empty-safe)."""

    SOURCE_NAME = "prollm"

    def load(self, path: Path) -> List[RankRecord]:
        data = self._read_json(path)
        if not data:
            return []

        entries = data.get("entries", [])
        if not entries:
            note = data.get("note", "")
            logger.info("ProLLM adapter: no entries available. %s", note)
            return []

        logger.info("ProLLM adapter: loaded %d entries from %s", len(entries), path.name)

        return self._entries_to_records(
            entries=entries,
            source_name=self.SOURCE_NAME,
            benchmark_name=data.get("benchmark_name", "llm_as_judge"),
            source_url=data.get("source_url", ""),
            snapshot_date=data.get("snapshot_date", ""),
        )
