"""
Ingestion adapter for NVIDIA Judge's Verdict leaderboard snapshots.
"""

import logging
from pathlib import Path
from typing import List

from recommender.ingestion.base import BaseAdapter
from recommender.schemas import RankRecord

logger = logging.getLogger(__name__)


class NvidiaAdapter(BaseAdapter):
    """Load rank records from the NVIDIA Judge's Verdict JSON snapshot."""

    SOURCE_NAME = "nvidia"

    def load(self, path: Path) -> List[RankRecord]:
        data = self._read_json(path)
        if not data:
            return []

        entries = data.get("entries", [])
        logger.info("NVIDIA adapter: loaded %d entries from %s", len(entries), path.name)

        return self._entries_to_records(
            entries=entries,
            source_name=self.SOURCE_NAME,
            benchmark_name=data.get("benchmark_name", "judges_verdict"),
            source_url=data.get("source_url", ""),
            snapshot_date=data.get("snapshot_date", ""),
        )
