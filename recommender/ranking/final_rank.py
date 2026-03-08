"""
Final lexicographic ranking and same-family filtering for the Judge recommendation flow.

Combines NVIDIA, ProLLM, and domain rank sources into a single ordering
using lexicographic comparison with missing-rank-last semantics.
"""

import logging
from typing import Dict, List, Optional, Tuple

from recommender.models import get_registry
from recommender.schemas import RankRecord, RecommendedJudge

logger = logging.getLogger(__name__)

# Sentinel for missing ranks -- sorts after any real rank
_INF = float("inf")


def _build_source_rank_maps(
    records: List[RankRecord],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build per-model rank dicts for nvidia and prollm sources.

    Returns (nvidia_ranks, prollm_ranks) each mapping
    canonical_model_name -> rank.
    """
    nvidia: Dict[str, int] = {}
    prollm: Dict[str, int] = {}

    for rec in records:
        name = rec.canonical_model_name or rec.model_name_raw
        if rec.source_name == "nvidia":
            nvidia[name] = rec.rank
        elif rec.source_name == "prollm":
            prollm[name] = rec.rank

    return nvidia, prollm


def _compute_coverage(
    model: str,
    nvidia_ranks: Dict[str, int],
    prollm_ranks: Dict[str, int],
    domain_ranks: Dict[str, float],
) -> Tuple[int, List[str]]:
    """Compute coverage count and list of source names for a model."""
    sources: List[str] = []
    if model in nvidia_ranks:
        sources.append("nvidia")
    if model in prollm_ranks:
        sources.append("prollm")
    if model in domain_ranks:
        sources.append("domain")
    return len(sources), sources


def build_candidate_pool(records: List[RankRecord]) -> List[str]:
    """Return deduplicated list of canonical model names across all records."""
    seen: Dict[str, None] = {}
    for rec in records:
        name = rec.canonical_model_name or rec.model_name_raw
        seen[name] = None
    return list(seen.keys())


def apply_same_family_filter(
    candidates: List[str],
    evaluated_model_family: Optional[str],
    exclude_same_family: bool,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Filter out candidates whose family matches the evaluated model's family.

    Returns:
        (filtered_candidates, excluded_list) where excluded_list contains
        (model_name, reason) tuples.
    """
    if not exclude_same_family or not evaluated_model_family:
        return candidates, []

    registry = get_registry()
    family_lower = evaluated_model_family.lower()
    kept: List[str] = []
    excluded: List[Tuple[str, str]] = []

    for model in candidates:
        model_family = registry.get_family(model).lower()
        if model_family == family_lower:
            reason = f"Same family as evaluated model ({evaluated_model_family})"
            excluded.append((model, reason))
            logger.info("Excluding '%s' (family=%s): same family filter", model, model_family)
        else:
            kept.append(model)

    return kept, excluded


def rank_candidates(
    candidates: List[str],
    nvidia_ranks: Dict[str, int],
    prollm_ranks: Dict[str, int],
    domain_ranks: Dict[str, float],
) -> List[RecommendedJudge]:
    """
    Sort candidates by lexicographic key and return ranked RecommendedJudge list.

    Sort key (ascending):
      1. nvidia_rank  (missing -> INF)
      2. prollm_rank  (missing -> INF)
      3. domain_rank  (missing -> INF)
      4. -coverage_count  (higher coverage first)
      5. model_name   (alphabetical tie-break)
    """
    registry = get_registry()

    def sort_key(model: str) -> Tuple[float, float, float, int, str]:
        nr = nvidia_ranks.get(model, _INF)
        pr = prollm_ranks.get(model, _INF)
        dr = domain_ranks.get(model, _INF)
        coverage, _ = _compute_coverage(model, nvidia_ranks, prollm_ranks, domain_ranks)
        return (nr, pr, dr, -coverage, model)

    sorted_candidates = sorted(candidates, key=sort_key)

    results: List[RecommendedJudge] = []
    for position, model in enumerate(sorted_candidates, start=1):
        cov_count, cov_sources = _compute_coverage(
            model, nvidia_ranks, prollm_ranks, domain_ranks
        )
        results.append(RecommendedJudge(
            model_name=model,
            model_family=registry.get_family(model),
            final_rank_position=position,
            nvidia_rank=nvidia_ranks.get(model),
            prollm_rank=prollm_ranks.get(model),
            domain_rank=domain_ranks.get(model),
            coverage_count=cov_count,
            coverage_sources=cov_sources,
        ))

    return results
