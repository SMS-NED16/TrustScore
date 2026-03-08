"""
Direct benchmark lookup for the Model recommendation flow.

Loads a single benchmark's rank file and returns its models in order,
with canonicalized names and metadata.
"""

import logging
from typing import Dict, List, Tuple

from recommender.canonicalization import canonicalize_records
from recommender.ingestion.domain_benchmarks import DomainBenchmarkAdapter
from recommender.models import get_registry
from recommender.schemas import RankRecord, RecommendedModel

logger = logging.getLogger(__name__)


def lookup_benchmark(
    benchmark_id: str,
    top_k: int = 5,
) -> Tuple[List[RecommendedModel], Dict[str, str]]:
    """
    Load a single benchmark's ranked models and return top-k.

    Returns:
        (recommended_models, file_metadata) where file_metadata has keys like
        metric_name, metric_units, metric_direction, snapshot_date, source_url.
    """
    adapter = DomainBenchmarkAdapter()
    records, file_meta = adapter.load_single_benchmark_with_metadata(benchmark_id)

    if not records:
        logger.warning("No records found for benchmark '%s'", benchmark_id)
        return [], file_meta

    canonicalize_records(records)

    # Sort by rank (should already be sorted, but be safe)
    records.sort(key=lambda r: r.rank)

    registry = get_registry()
    results: List[RecommendedModel] = []
    for rec in records[:top_k]:
        canonical = rec.canonical_model_name or rec.model_name_raw
        results.append(RecommendedModel(
            model_name=canonical,
            model_family=registry.get_family(canonical),
            rank=rec.rank,
            score=rec.score,
            benchmark_name=file_meta.get("benchmark_name", benchmark_id),
            metric_name=file_meta.get("metric_name", ""),
            metric_units=file_meta.get("metric_units", ""),
            metric_direction=file_meta.get("metric_direction", "higher_better"),
            snapshot_date=file_meta.get("snapshot_date", ""),
        ))

    logger.info(
        "Benchmark lookup '%s': returning %d of %d models",
        benchmark_id, len(results), len(records),
    )
    return results, file_meta
