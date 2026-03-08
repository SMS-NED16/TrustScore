"""
Pydantic models for the TrustScore Recommender.

Defines data structures for rank records, model identities, taxonomy nodes,
and request/response schemas for both the Judge and Model recommendation flows.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class RankRecord(BaseModel):
    """A single rank entry from any leaderboard source."""
    source_name: str = Field(..., description="Origin source: 'nvidia', 'prollm', or 'domain'")
    benchmark_name: str = Field(..., description="Benchmark identifier, e.g. 'facts_suite'")
    model_name_raw: str = Field(..., description="Model name as it appears in the source file")
    canonical_model_name: Optional[str] = Field(default=None, description="Resolved canonical model name")
    rank: int = Field(..., ge=1, description="Rank position (1 = best)")
    score: Optional[float] = Field(default=None, description="Original metric score (for display only)")
    source_url: Optional[str] = Field(default=None)
    snapshot_date: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelIdentity(BaseModel):
    """Canonical representation of a model in the registry."""
    canonical_name: str
    display_name: str
    family: str
    aliases: List[str] = Field(default_factory=list)


class BenchmarkMetadata(BaseModel):
    """Metadata about a benchmark (metric info, snapshot date, source)."""
    benchmark_name: str
    metric_name: str = ""
    metric_units: str = ""
    metric_direction: str = "higher_better"
    source_url: str = ""
    snapshot_date: str = ""


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

class TaxonomyLeaf(BaseModel):
    """A leaf node in the taxonomy -- maps to exactly one benchmark."""
    display_name: str
    benchmark: str
    metric_name: str = ""
    metric_units: str = ""
    metric_direction: str = "higher_better"


class TaxonomyBranch(BaseModel):
    """A branch node in the taxonomy -- has children but no direct benchmark."""
    display_name: str
    children: Dict[str, Any] = Field(default_factory=dict)


class TaxonomyNodeResponse(BaseModel):
    """API response when resolving a taxonomy path."""
    path: str
    display_name: str
    is_leaf: bool
    benchmark: Optional[str] = None
    metric_name: Optional[str] = None
    metric_units: Optional[str] = None
    metric_direction: Optional[str] = None
    children: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Judge recommendation flow
# ---------------------------------------------------------------------------

class JudgeDomainConfig(BaseModel):
    """Configuration for a flat judge-flow domain alias."""
    display_name: str
    benchmarks: List[str] = Field(default_factory=list)


class JudgeRecommendationRequest(BaseModel):
    """Input for the judge recommendation flow."""
    domain: str
    top_k: int = Field(default=3, ge=1)
    evaluated_model_name: Optional[str] = None
    evaluated_model_family: Optional[str] = None
    exclude_same_family: bool = True


class RecommendedJudge(BaseModel):
    """A single judge recommendation row."""
    model_name: str
    model_family: str
    final_rank_position: int
    nvidia_rank: Optional[int] = None
    prollm_rank: Optional[int] = None
    domain_rank: Optional[float] = None
    coverage_count: int = 0
    coverage_sources: List[str] = Field(default_factory=list)
    excluded: bool = False
    exclusion_reason: Optional[str] = None


class JudgeRecommendationResponse(BaseModel):
    """Output of the judge recommendation flow."""
    domain: str
    top_k: int
    candidate_pool_size: int = 0
    used_domain_benchmarks: List[str] = Field(default_factory=list)
    recommended_models: List[RecommendedJudge] = Field(default_factory=list)
    decision_trace: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model recommendation flow
# ---------------------------------------------------------------------------

class ModelRecommendationRequest(BaseModel):
    """Input for the model recommendation flow."""
    taxonomy_path: str
    top_k: int = Field(default=5, ge=1)


class RecommendedModel(BaseModel):
    """A single model recommendation row from direct benchmark lookup."""
    model_name: str
    model_family: str
    rank: int
    score: Optional[float] = None
    benchmark_name: str = ""
    metric_name: str = ""
    metric_units: str = ""
    metric_direction: str = "higher_better"
    snapshot_date: str = ""


class ModelRecommendationResponse(BaseModel):
    """Output of the model recommendation flow."""
    taxonomy_path: str
    display_name: str
    benchmark_name: str
    metric_name: str = ""
    metric_units: str = ""
    metric_direction: str = "higher_better"
    snapshot_date: str = ""
    source_url: str = ""
    top_k: int
    recommended_models: List[RecommendedModel] = Field(default_factory=list)
