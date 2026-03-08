"""
Ingestion adapter for domain-specific benchmark rank files.

Each benchmark is stored as a separate JSON file under ranks/domain/.
This adapter can load one or many benchmark files.
"""

import logging
from pathlib import Path
from typing import Dict, List

from recommender.config import get_rank_file_path
from recommender.ingestion.base import BaseAdapter
from recommender.schemas import BenchmarkMetadata, RankRecord

logger = logging.getLogger(__name__)


class DomainBenchmarkAdapter(BaseAdapter):
    """Load rank records from one or more domain benchmark JSON files."""

    SOURCE_NAME = "domain"

    def load(self, path: Path) -> List[RankRecord]:
        """Load a single benchmark file."""
        data = self._read_json(path)
        if not data:
            return []

        entries = data.get("entries", [])
        benchmark_name = data.get("benchmark_name", path.stem)
        logger.info(
            "Domain adapter: loaded %d entries for benchmark '%s' from %s",
            len(entries), benchmark_name, path.name,
        )

        return self._entries_to_records(
            entries=entries,
            source_name=self.SOURCE_NAME,
            benchmark_name=benchmark_name,
            source_url=data.get("source_url", ""),
            snapshot_date=data.get("snapshot_date", ""),
        )

    def load_benchmarks(self, benchmark_ids: List[str]) -> List[RankRecord]:
        """
        Load rank records for multiple benchmark IDs.

        Resolves each ID to a file path via config and loads them.
        Returns a combined list of all records.
        """
        all_records: List[RankRecord] = []
        for bid in benchmark_ids:
            path = get_rank_file_path(bid)
            records = self.load(path)
            all_records.extend(records)
        return all_records

    def load_single_benchmark_with_metadata(
        self, benchmark_id: str
    ) -> tuple[List[RankRecord], Dict[str, str]]:
        """
        Load a single benchmark's records plus its file-level metadata dict.

        Returns (records, metadata_dict) where metadata_dict has keys like
        metric_name, metric_units, metric_direction, snapshot_date, source_url.
        """
        path = get_rank_file_path(benchmark_id)
        data = self._read_json(path)
        if not data:
            return [], {}

        entries = data.get("entries", [])
        benchmark_name = data.get("benchmark_name", benchmark_id)

        records = self._entries_to_records(
            entries=entries,
            source_name=self.SOURCE_NAME,
            benchmark_name=benchmark_name,
            source_url=data.get("source_url", ""),
            snapshot_date=data.get("snapshot_date", ""),
        )

        file_meta = {
            "benchmark_name": benchmark_name,
            "metric_name": data.get("metric_name", ""),
            "metric_units": data.get("metric_units", ""),
            "metric_direction": data.get("metric_direction", "higher_better"),
            "snapshot_date": data.get("snapshot_date", ""),
            "source_url": data.get("source_url", ""),
        }

        return records, file_meta
