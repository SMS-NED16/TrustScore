"""
Domain rank computation for the Judge recommendation flow.

Given a set of benchmark IDs for a domain, computes a single per-model
"domain rank" using the median of that model's ranks across the domain's
benchmarks, with tie-breaking by coverage and best individual rank.
"""

import logging
import statistics
from typing import Dict, List, Optional, Tuple

from recommender.schemas import RankRecord

logger = logging.getLogger(__name__)


def compute_domain_ranks(
    records: List[RankRecord],
    benchmark_ids: List[str],
) -> Dict[str, float]:
    """
    Compute a single domain rank per canonical model.

    Steps:
      1. Group records by canonical_model_name, keeping only those whose
         benchmark_name is in *benchmark_ids*.
      2. For each model, compute the **median** rank across its benchmarks.
      3. Tie-break:
         a. Higher benchmark coverage (more benchmarks with a rank).
         b. Best (lowest) individual benchmark rank.
      4. Sort models by (median_rank, -coverage, best_rank) ascending.
      5. Assign sequential domain rank positions starting at 1.

    Returns:
        dict mapping canonical_model_name -> domain rank (float, 1-indexed).
    """
    if not records or not benchmark_ids:
        return {}

    bid_set = set(benchmark_ids)

    # Step 1: group ranks by model
    model_ranks: Dict[str, List[int]] = {}
    for rec in records:
        if rec.benchmark_name not in bid_set:
            continue
        name = rec.canonical_model_name or rec.model_name_raw
        model_ranks.setdefault(name, []).append(rec.rank)

    if not model_ranks:
        logger.info("No records matched domain benchmarks %s", benchmark_ids)
        return {}

    # Step 2-3: compute sort key per model
    sort_entries: List[Tuple[float, int, int, str]] = []
    for model, ranks in model_ranks.items():
        median = statistics.median(ranks)
        coverage = len(ranks)
        best = min(ranks)
        # Sort ascending: lower median first; for ties, higher coverage first
        # (negate coverage so ascending sort puts higher coverage first),
        # then lower best rank first, then alphabetical.
        sort_entries.append((median, -coverage, best, model))

    sort_entries.sort()

    # Step 5: assign domain rank positions
    domain_ranks: Dict[str, float] = {}
    for position, (median, neg_cov, best, model) in enumerate(sort_entries, start=1):
        domain_ranks[model] = float(position)
        logger.debug(
            "Domain rank %d: model=%s  median=%.1f  coverage=%d  best=%d",
            position, model, median, -neg_cov, best,
        )

    return domain_ranks
