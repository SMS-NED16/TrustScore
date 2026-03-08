"""
Service layer for the TrustScore Recommender.

Exposes two public functions:
  - recommend_judges()  -- multi-source rank aggregation for judge selection
  - recommend_model()   -- direct benchmark lookup via hierarchical taxonomy
"""

import logging
from typing import Dict, List, Optional

from recommender.canonicalization import canonicalize_records
from recommender.config import (
    get_judge_domains,
    get_rank_file_path,
    resolve_taxonomy_path,
)
from recommender.ingestion.domain_benchmarks import DomainBenchmarkAdapter
from recommender.ingestion.nvidia import NvidiaAdapter
from recommender.ingestion.prollm import ProllmAdapter
from recommender.models import get_registry
from recommender.ranking.benchmark_lookup import lookup_benchmark
from recommender.ranking.domain_rank import compute_domain_ranks
from recommender.ranking.final_rank import (
    apply_same_family_filter,
    build_candidate_pool,
    rank_candidates,
    _build_source_rank_maps,
)
from recommender.schemas import (
    JudgeRecommendationResponse,
    ModelRecommendationResponse,
    RankRecord,
    RecommendedJudge,
    TaxonomyNodeResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge recommendation flow
# ---------------------------------------------------------------------------

def _validate_judge_domain(domain: str) -> List[str]:
    """
    Validate that *domain* is a known judge-flow domain alias.

    Returns the list of benchmark IDs for that domain.
    Raises ValueError if the domain is not recognized.
    """
    domains = get_judge_domains()
    if domain not in domains:
        available = sorted(domains.keys())
        raise ValueError(
            f"Unknown domain '{domain}'. Available domains: {available}"
        )
    return domains[domain].benchmarks


def _load_all_rank_records(benchmark_ids: List[str]) -> List[RankRecord]:
    """
    Load rank records from all three sources (NVIDIA, ProLLM, domain benchmarks)
    and canonicalize all model names.
    """
    records: List[RankRecord] = []

    # NVIDIA
    nvidia_path = get_rank_file_path("nvidia_judges_verdict")
    nvidia_records = NvidiaAdapter().load(nvidia_path)
    records.extend(nvidia_records)

    # ProLLM
    prollm_path = get_rank_file_path("prollm_judge")
    prollm_records = ProllmAdapter().load(prollm_path)
    records.extend(prollm_records)

    # Domain benchmarks
    if benchmark_ids:
        domain_records = DomainBenchmarkAdapter().load_benchmarks(benchmark_ids)
        records.extend(domain_records)

    canonicalize_records(records)
    logger.info(
        "Loaded %d total rank records (nvidia=%d, prollm=%d, domain=%d)",
        len(records), len(nvidia_records), len(prollm_records),
        len(records) - len(nvidia_records) - len(prollm_records),
    )
    return records


def _resolve_evaluated_family(
    evaluated_model_name: Optional[str],
    evaluated_model_family: Optional[str],
) -> Optional[str]:
    """
    Determine the evaluated model's family.
    Explicit family overrides; otherwise infer from model name via registry.
    """
    if evaluated_model_family:
        return evaluated_model_family
    if evaluated_model_name:
        family = get_registry().get_family(evaluated_model_name)
        if family != "unknown":
            logger.info(
                "Inferred family '%s' from evaluated_model_name '%s'",
                family, evaluated_model_name,
            )
            return family
    return None


def recommend_judges(
    domain: str,
    top_k: int = 3,
    evaluated_model_name: Optional[str] = None,
    evaluated_model_family: Optional[str] = None,
    exclude_same_family: bool = True,
) -> Dict:
    """
    Recommend top-k judge models for a given domain.

    Args:
        domain: Judge-flow domain alias (e.g. 'summarization', 'general_qa').
        top_k: Number of judges to recommend.
        evaluated_model_name: Optional name of the model being evaluated.
        evaluated_model_family: Optional family override for same-family exclusion.
        exclude_same_family: Whether to exclude judges from the same family.

    Returns:
        Dict matching JudgeRecommendationResponse schema.

    Raises:
        ValueError: If domain is invalid or top_k < 1.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    # Step 1: Validate domain and get benchmark list
    benchmark_ids = _validate_judge_domain(domain)
    trace: List[str] = [f"Domain '{domain}' maps to benchmarks: {benchmark_ids}"]

    # Step 2: Load all rank records
    all_records = _load_all_rank_records(benchmark_ids)
    trace.append(f"Loaded {len(all_records)} total rank records")

    # Step 3: Build source-specific rank maps
    nvidia_ranks, prollm_ranks = _build_source_rank_maps(all_records)
    trace.append(f"NVIDIA ranks for {len(nvidia_ranks)} models, ProLLM ranks for {len(prollm_ranks)} models")

    # Step 4: Compute domain rank
    domain_rank_map = compute_domain_ranks(all_records, benchmark_ids)
    trace.append(f"Domain ranks computed for {len(domain_rank_map)} models")

    # Step 5: Build candidate pool
    candidates = build_candidate_pool(all_records)
    pool_size = len(candidates)
    trace.append(f"Candidate pool: {pool_size} models")

    # Step 6: Same-family exclusion
    family = _resolve_evaluated_family(evaluated_model_name, evaluated_model_family)
    candidates, excluded = apply_same_family_filter(candidates, family, exclude_same_family)
    if excluded:
        trace.append(f"Excluded {len(excluded)} model(s) by same-family filter: "
                      f"{[e[0] for e in excluded]}")

    # Step 7: Rank remaining candidates
    ranked = rank_candidates(candidates, nvidia_ranks, prollm_ranks, domain_rank_map)

    # Step 8: Take top-k
    top = ranked[:top_k]
    trace.append(f"Returning top {len(top)} of {len(ranked)} ranked candidates")

    response = JudgeRecommendationResponse(
        domain=domain,
        top_k=top_k,
        candidate_pool_size=pool_size,
        used_domain_benchmarks=benchmark_ids,
        recommended_models=top,
        decision_trace=trace,
    )

    logger.info(
        "recommend_judges(domain=%s, top_k=%d): returning %d models from pool of %d",
        domain, top_k, len(top), pool_size,
    )
    return response.model_dump()


# ---------------------------------------------------------------------------
# Model recommendation flow
# ---------------------------------------------------------------------------

def recommend_model(
    taxonomy_path: str,
    top_k: int = 5,
) -> Dict:
    """
    Recommend top-k models for a specific task via direct benchmark lookup.

    The taxonomy_path must resolve to a leaf node (which maps to a benchmark).
    If it resolves to a branch, a helpful error with available children is returned.

    Args:
        taxonomy_path: Dot-delimited path in taxonomy (e.g.
                       'trustworthiness.factuality.multi_scenario').
        top_k: Number of models to recommend.

    Returns:
        Dict matching ModelRecommendationResponse schema.

    Raises:
        ValueError: If path is invalid or not a leaf.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    # Resolve taxonomy node
    node = resolve_taxonomy_path(taxonomy_path)

    if not node.is_leaf:
        children_desc = ", ".join(
            f"'{c['path']}' ({c['display_name']})"
            for c in (node.children or [])
        )
        raise ValueError(
            f"Taxonomy path '{taxonomy_path}' is a branch, not a leaf. "
            f"Drill down into one of: {children_desc}"
        )

    benchmark_id = node.benchmark
    if not benchmark_id:
        raise ValueError(f"Leaf node at '{taxonomy_path}' has no benchmark mapping.")

    # Direct benchmark lookup
    recommended, file_meta = lookup_benchmark(benchmark_id, top_k=top_k)

    response = ModelRecommendationResponse(
        taxonomy_path=taxonomy_path,
        display_name=node.display_name or taxonomy_path,
        benchmark_name=file_meta.get("benchmark_name", benchmark_id),
        metric_name=file_meta.get("metric_name", ""),
        metric_units=file_meta.get("metric_units", ""),
        metric_direction=file_meta.get("metric_direction", "higher_better"),
        snapshot_date=file_meta.get("snapshot_date", ""),
        source_url=file_meta.get("source_url", ""),
        top_k=top_k,
        recommended_models=recommended,
    )

    logger.info(
        "recommend_model(path=%s, top_k=%d): returning %d models from benchmark '%s'",
        taxonomy_path, top_k, len(recommended), benchmark_id,
    )
    return response.model_dump()
