"""
Configuration loader for the TrustScore Recommender.

Loads taxonomy.yaml and models.yaml from recommender/data/ and provides
helpers for taxonomy navigation, judge-domain lookup, and rank file path resolution.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from recommender.schemas import (
    BenchmarkMetadata,
    JudgeDomainConfig,
    ModelIdentity,
    TaxonomyNodeResponse,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
RANKS_DIR = DATA_DIR / "ranks"
DOMAIN_RANKS_DIR = RANKS_DIR / "domain"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Singleton-style cached config
# ---------------------------------------------------------------------------

_taxonomy_cache: Optional[Dict[str, Any]] = None
_judge_domains_cache: Optional[Dict[str, JudgeDomainConfig]] = None
_models_cache: Optional[List[ModelIdentity]] = None


def _load_taxonomy_file() -> Dict[str, Any]:
    return _load_yaml(DATA_DIR / "taxonomy.yaml")


def get_taxonomy_tree() -> Dict[str, Any]:
    """Return the full hierarchical taxonomy dict (cached)."""
    global _taxonomy_cache
    if _taxonomy_cache is None:
        raw = _load_taxonomy_file()
        _taxonomy_cache = raw.get("taxonomy", {})
        logger.info("Loaded taxonomy tree with %d top-level domains", len(_taxonomy_cache))
    return _taxonomy_cache


def get_judge_domains() -> Dict[str, JudgeDomainConfig]:
    """Return the flat judge-domain aliases (cached)."""
    global _judge_domains_cache
    if _judge_domains_cache is None:
        raw = _load_taxonomy_file()
        raw_domains = raw.get("judge_domains", {})
        _judge_domains_cache = {
            key: JudgeDomainConfig(**val)
            for key, val in raw_domains.items()
        }
        logger.info("Loaded %d judge domain aliases", len(_judge_domains_cache))
    return _judge_domains_cache


def get_model_definitions() -> List[ModelIdentity]:
    """Return the model registry list (cached)."""
    global _models_cache
    if _models_cache is None:
        raw = _load_yaml(DATA_DIR / "models.yaml")
        _models_cache = [ModelIdentity(**m) for m in raw.get("models", [])]
        logger.info("Loaded %d model definitions", len(_models_cache))
    return _models_cache


def invalidate_caches() -> None:
    """Clear all cached config (useful for testing)."""
    global _taxonomy_cache, _judge_domains_cache, _models_cache
    _taxonomy_cache = None
    _judge_domains_cache = None
    _models_cache = None


# ---------------------------------------------------------------------------
# Taxonomy navigation
# ---------------------------------------------------------------------------

def _is_leaf(node: Dict[str, Any]) -> bool:
    """A leaf node has a 'benchmark' key."""
    return "benchmark" in node


def resolve_taxonomy_path(path: str) -> TaxonomyNodeResponse:
    """
    Walk the taxonomy tree along a dot-delimited path and return the node info.

    - Leaf node: returns benchmark details.
    - Branch node: returns list of children with their display names.
    - Invalid path: raises ValueError.
    """
    tree = get_taxonomy_tree()
    segments = [s for s in path.split(".") if s]

    current: Dict[str, Any] = tree
    display_parts: List[str] = []

    for i, seg in enumerate(segments):
        if seg not in current:
            raise ValueError(
                f"Invalid taxonomy path: '{path}' — segment '{seg}' not found "
                f"at level {i}. Available keys: {list(current.keys())}"
            )
        current = current[seg]
        display_parts.append(current.get("display_name", seg))

        # If this is a branch, descend into its children for the next segment
        if not _is_leaf(current) and "children" in current:
            if i < len(segments) - 1:
                current = current["children"]

    display_name = " > ".join(display_parts) if display_parts else "Root"

    if _is_leaf(current):
        return TaxonomyNodeResponse(
            path=path,
            display_name=display_name,
            is_leaf=True,
            benchmark=current.get("benchmark"),
            metric_name=current.get("metric_name"),
            metric_units=current.get("metric_units"),
            metric_direction=current.get("metric_direction"),
        )

    # Branch node — enumerate children
    children_dict = current.get("children", current)
    children_list = []
    for key, child in children_dict.items():
        if isinstance(child, dict):
            children_list.append({
                "id": key,
                "display_name": child.get("display_name", key),
                "is_leaf": _is_leaf(child),
                "path": f"{path}.{key}" if path else key,
            })

    return TaxonomyNodeResponse(
        path=path,
        display_name=display_name,
        is_leaf=False,
        children=children_list,
    )


# ---------------------------------------------------------------------------
# Rank file resolution
# ---------------------------------------------------------------------------

def get_rank_file_path(benchmark_id: str) -> Path:
    """
    Resolve a benchmark identifier to its rank JSON file path.

    Convention:
      - nvidia_judges_verdict, prollm_judge -> ranks/<id>.json
      - everything else -> ranks/domain/<id>.json
    """
    top_level = {"nvidia_judges_verdict", "prollm_judge"}
    if benchmark_id in top_level:
        return RANKS_DIR / f"{benchmark_id}.json"
    return DOMAIN_RANKS_DIR / f"{benchmark_id}.json"


def get_benchmark_metadata(benchmark_id: str) -> Optional[BenchmarkMetadata]:
    """
    Load just the metadata fields from a rank JSON file
    (benchmark_name, metric_*, source_url, snapshot_date).
    """
    import json

    path = get_rank_file_path(benchmark_id)
    if not path.exists():
        logger.warning("Rank file not found for benchmark '%s' at %s", benchmark_id, path)
        return None

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return BenchmarkMetadata(
        benchmark_name=data.get("benchmark_name", benchmark_id),
        metric_name=data.get("metric_name", ""),
        metric_units=data.get("metric_units", ""),
        metric_direction=data.get("metric_direction", "higher_better"),
        source_url=data.get("source_url", ""),
        snapshot_date=data.get("snapshot_date", ""),
    )
