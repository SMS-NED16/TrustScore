"""
Model name canonicalization for the TrustScore Recommender.

Maps raw model names from leaderboard files to canonical model IDs and families
using the ModelRegistry.
"""

import logging
from typing import List

from recommender.models import get_registry
from recommender.schemas import RankRecord

logger = logging.getLogger(__name__)


def canonicalize_record(record: RankRecord) -> RankRecord:
    """
    Resolve the raw model name on a RankRecord to its canonical name.
    Mutates and returns the same record for convenience.
    """
    registry = get_registry()
    canonical = registry.get_canonical_name(record.model_name_raw)

    if canonical != record.model_name_raw:
        logger.debug(
            "Canonicalized '%s' -> '%s' (source=%s, benchmark=%s)",
            record.model_name_raw, canonical,
            record.source_name, record.benchmark_name,
        )
    else:
        logger.debug(
            "No alias found for '%s'; using raw name as canonical (source=%s)",
            record.model_name_raw, record.source_name,
        )

    record.canonical_model_name = canonical
    return record


def canonicalize_records(records: List[RankRecord]) -> List[RankRecord]:
    """Canonicalize a batch of RankRecords in place."""
    for record in records:
        canonicalize_record(record)
    return records
